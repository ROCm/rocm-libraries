// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

/// Build an SDPA forward graph with explicit per-tensor input/output data types and
/// optional per-tensor Q/K/V descale tensors. Used to exercise the FP8 + descale
/// paths of SdpaFwdPlanBuilder/SdpaFwdPlan on the CPU (no GPU kernels involved).
/// Fixed UIDs: Q=1, K=2, V=3, O=4, descaleQ=5, descaleK=6, descaleV=7.
std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildDescaleSdpaFwdGraph(const std::vector<int64_t>& qDims,
                             const std::vector<int64_t>& kDims,
                             const std::vector<int64_t>& vDims,
                             hipdnn_frontend::DataType inputDataType,
                             hipdnn_frontend::DataType outputDataType,
                             bool withDescale,
                             bool withDescaleS = false)
{
    using hipdnn_data_sdk::utilities::generateStrides;
    using hipdnn_frontend::DataType;
    using hipdnn_frontend::graph::Graph;
    using hipdnn_frontend::graph::SdpaAttributes;
    using hipdnn_frontend::graph::TensorAttributes;

    auto makeAttr = [](const std::string& name,
                       DataType dataType,
                       const std::vector<int64_t>& dims,
                       int64_t uid) {
        auto attr = std::make_shared<TensorAttributes>();
        attr->set_name(name)
            .set_data_type(dataType)
            .set_dim(dims)
            .set_stride(generateStrides(dims))
            .set_uid(uid);
        return attr;
    };

    auto graph = std::make_shared<Graph>();
    graph->set_name("SdpaFwdDescaleTest");
    graph->set_io_data_type(inputDataType)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto qAttr = makeAttr("Q", inputDataType, qDims, 1);
    auto kAttr = makeAttr("K", inputDataType, kDims, 2);
    auto vAttr = makeAttr("V", inputDataType, vDims, 3);

    SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("SdpaFwdDescale");
    if(withDescale)
    {
        const std::vector<int64_t> scalarDims = {1, 1, 1, 1};
        sdpaAttrs.set_descale_q(makeAttr("DESCALE_Q", DataType::FLOAT, scalarDims, 5));
        sdpaAttrs.set_descale_k(makeAttr("DESCALE_K", DataType::FLOAT, scalarDims, 6));
        sdpaAttrs.set_descale_v(makeAttr("DESCALE_V", DataType::FLOAT, scalarDims, 7));
        if(withDescaleS)
        {
            // Softmax descale is part of the fp8 contract but unsupported by the v3
            // forward kernel, so the executor must reject graphs that carry it.
            sdpaAttrs.set_descale_s(makeAttr("DESCALE_S", DataType::FLOAT, scalarDims, 8));
        }
    }

    auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, sdpaAttrs);
    if(!oAttr->has_uid())
    {
        oAttr->set_uid(4);
    }
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};
    oAttr->set_data_type(outputDataType)
        .set_dim(oDims)
        .set_stride(generateStrides(oDims))
        .set_is_virtual(false);

    return graph;
}

} // namespace

namespace
{

// Builds a minimal single-node SDPA-forward flatbuffer graph directly. The frontend
// graph builder only emits the O and Stats (LSE) outputs and silently drops any
// max/sum_exp request, so the plan builder's rejection of those softmax-stats outputs
// can only be exercised at the flatbuffer level. `attrs` carries the unsupported field
// under test; the q/k/v/o uids and a packed float [1, 2, 4, 8] shape are filled in here.
flatbuffers::FlatBufferBuilder makeRawSdpaFwdGraph(SdpaAttributesT attrs)
{
    constexpr int64_t Q_UID = 1;
    constexpr int64_t K_UID = 2;
    constexpr int64_t V_UID = 3;
    constexpr int64_t O_UID = 4;
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    attrs.q_tensor_uid = Q_UID;
    attrs.k_tensor_uid = K_UID;
    attrs.v_tensor_uid = V_UID;
    attrs.o_tensor_uid = O_UID;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, Q_UID, "Q", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, K_UID, "K", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, V_UID, "V", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, O_UID, "O", DataType::FLOAT, &strides, &dims));

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(builder,
                                   "SdpaFwdRejectGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace

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
                         /*leftBound=*/-1,
                         /*rightBound=*/-1,
                         /*topLeftAlignment=*/true);

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

TEST(TestSdpaFwdPlan, ExecutePlanWithRuntimeScaleFromPack)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    // Pure runtime pass-by-value scale delivered through the variant pack.
    float scaleHostValue = 0.25f;
    auto graphTuple = buildSdpaFwdGraph(planTensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/false,
                                        /*generateStats=*/false,
                                        /*runtimeScaleHostPtr=*/&scaleHostValue);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_TRUE(nodeAttributes->scale_tensor_uid().has_value());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        builder;
    auto builtPlan = builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    builtPlan->execute(variantPack);

    // Direct reference with the same explicit scale value.
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            scaleHostValue);

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
                         /*leftBound=*/-1,
                         /*rightBound=*/0,
                         /*topLeftAlignment=*/true);

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

TEST(TestSdpaFwdPlanBuilder, ExecutePlanWithAsymmetricWindow)
{
    // Regression test for a bug where SdpaFwdPlanBuilder forwarded the same value
    // (left_bound) for both leftBound and rightBound to CpuFpReferenceSdpa::forward,
    // ignoring right_bound. With leftBound == rightBound (e.g. for symmetric windows
    // or causal masks where both are -1/0), that bug is invisible. This test uses an
    // asymmetric window (leftBound=2, rightBound=1) so that any confusion of the two
    // bounds produces a different attention pattern and a divergent output.
    //
    // Compares the output of the dispatched plan (graph → SdpaFwdPlanBuilder →
    // SdpaFwdPlan::execute) against a direct call to CpuFpReferenceSdpa::forward
    // with the same asymmetric window parameters.
    //
    // [B=1, H=2, Sq=4, Skv=4, D=8] with leftBound=2, rightBound=1, TopLeft alignment.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    constexpr int64_t LEFT_BOUND = 2;
    constexpr int64_t RIGHT_BOUND = 1;

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/LEFT_BOUND,
                                        /*rightBound=*/RIGHT_BOUND,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    // Build the plan through SdpaFwdPlanBuilder so the dispatcher's left/right bound
    // extraction is exercised (this is where the original bug lived).
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    auto plan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();
    plan->execute(variantPack);

    // Direct CPU reference with the same asymmetric window parameters.
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            std::nullopt,
                                                            noMask,
                                                            LEFT_BOUND,
                                                            RIGHT_BOUND,
                                                            /*topLeftAlignment=*/true);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor))
        << "Plan output (via SdpaFwdPlanBuilder) does not match direct CpuFpReferenceSdpa "
           "with leftBound="
        << LEFT_BOUND << ", rightBound=" << RIGHT_BOUND
        << ". This indicates the dispatcher is not forwarding the two bounds distinctly.";
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

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsAlibiMask)
{
    // SdpaFwdPlanBuilder does not implement ALiBi positional encoding, so it must
    // refuse to claim nodes that have alibi_mask=true. Otherwise a graph with ALiBi
    // would silently fall through to a CPU plan that ignores the ALiBi attribute
    // and produces wrong results.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
        << "SdpaFwdPlanBuilder must reject nodes with alibi_mask=true";
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsMaxStats)
{
    // The reference produces only the log-sum-exp stats output; the running-max softmax
    // stat is not produced, so a node requesting max_tensor_uid must be rejected. (The
    // stats/LSE output, by contrast, is supported — see IsApplicableAndExecutesWithStatsOutput.)
    SdpaAttributesT attrs;
    attrs.max_tensor_uid = 50;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs);

    const GraphWrapper graphWrapper(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
        << "SdpaFwdPlanBuilder must reject nodes requesting the max softmax-stats output";
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsSumExpStats)
{
    // The running sum-exp softmax stat is not produced by the reference, so a node
    // requesting sum_exp_tensor_uid must be rejected (only the LSE stats output is supported).
    SdpaAttributesT attrs;
    attrs.sum_exp_tensor_uid = 51;
    auto graphBuilder = makeRawSdpaFwdGraph(attrs);

    const GraphWrapper graphWrapper(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
        << "SdpaFwdPlanBuilder must reject nodes requesting the sum_exp softmax-stats output";
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableAndExecutesWithStatsOutput)
{
    // A graph configured to emit the softmax log-sum-exp (LSE) stats output is supported.
    // The plan must be applicable, build into the concrete SdpaFwdPlan, report the stats uid
    // among its output tensor ids, and write rank-4 [B, H, Sq, 1] LSE values when executed.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/false,
                                        /*generateStats=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto& variantPack = std::get<1>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    ASSERT_TRUE(nodeAttributes->stats_tensor_uid().has_value());
    const int64_t statsUid = nodeAttributes->stats_tensor_uid().value();

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_TRUE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto builtPlan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    auto* concretePlan = dynamic_cast<SdpaFwdPlan<float, float, float, float>*>(builtPlan.get());
    ASSERT_NE(concretePlan, nullptr);

    // The stats uid must be reported as an output so the harness binds and validates it.
    const auto outputIds = concretePlan->getOutputTensorIds();
    EXPECT_NE(std::find(outputIds.begin(), outputIds.end(), statsUid), outputIds.end());

    // Allocate the rank-4 LSE output buffer and bind it before executing.
    hipdnn_data_sdk::utilities::Tensor<float> lseTensor({qDims[0], qDims[1], qDims[2], 1});
    lseTensor.fillWithValue(-1.0f);
    variantPack[statsUid] = lseTensor.memory().hostData();

    builtPlan->execute(variantPack);

    // Independently compute the expected LSE via the direct reference and compare.
    SdpaFwdTensorBundle<float> directBundle(qDims, kDims, vDims, /*seed=*/1);
    hipdnn_data_sdk::utilities::Tensor<float> directLse({qDims[0], qDims[1], qDims[2], 1});
    CpuFpReferenceSdpa::forward<float, float, float, float, float>(directBundle.qTensor,
                                                                   directBundle.kTensor,
                                                                   directBundle.vTensor,
                                                                   directBundle.oTensor,
                                                                   std::nullopt,
                                                                   /*attnMask=*/nullptr,
                                                                   /*leftBound=*/-1,
                                                                   /*rightBound=*/-1,
                                                                   /*topLeftAlignment=*/true,
                                                                   &directLse);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> lseValidation(tolerance, tolerance);
    EXPECT_TRUE(lseValidation.allClose(directLse, lseTensor))
        << "Plan-produced LSE does not match the direct CpuFpReferenceSdpa LSE output.";
}

TEST(TestSdpaFwdPlanBuilder, DeprecatedCausalMaskMatchesExplicitTopLeftBounds)
{
    // The dispatcher in SdpaFwdPlanBuilder maps the deprecated causal_mask=true flag
    // to (leftBound=-1, rightBound=0, TOP_LEFT). This test verifies that mapping by
    // running two graphs that should produce bit-for-bit identical output:
    //   (a) causal_mask=true                         (deprecated path)
    //   (b) leftBound=-1, rightBound=0, TOP_LEFT     (modern path)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> explicitBundle(qDims, kDims, vDims, seed);

    // (a) Deprecated causal_mask=true
    auto deprecatedGraphTuple = buildSdpaFwdGraph(deprecatedBundle,
                                                  DataType::FLOAT,
                                                  /*causalMask=*/true);
    auto& deprecatedGraph = std::get<0>(deprecatedGraphTuple);
    auto [depBin, depErr] = deprecatedGraph->to_binary();
    ASSERT_TRUE(depErr.is_good()) << depErr.get_message();
    const GraphWrapper depWrapper(depBin.data(), depBin.size());

    // (b) Explicit (leftBound=-1, rightBound=0, TOP_LEFT)
    auto explicitGraphTuple = buildSdpaFwdGraph(explicitBundle,
                                                DataType::FLOAT,
                                                /*causalMask=*/false,
                                                /*causalMaskBottomRight=*/false,
                                                /*leftBound=*/-1,
                                                /*rightBound=*/0,
                                                hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& explicitGraph = std::get<0>(explicitGraphTuple);
    auto [expBin, expErr] = explicitGraph->to_binary();
    ASSERT_TRUE(expErr.is_good()) << expErr.get_message();
    const GraphWrapper expWrapper(expBin.data(), expBin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    // Execute deprecated-path plan
    {
        auto plan = planBuilder.buildNodePlan(depWrapper, depWrapper.getNode(0));
        const auto* attrs = depWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = deprecatedBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = deprecatedBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = deprecatedBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = deprecatedBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Execute explicit-bounds plan
    {
        auto plan = planBuilder.buildNodePlan(expWrapper, expWrapper.getNode(0));
        const auto* attrs = expWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = explicitBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = explicitBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = explicitBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = explicitBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Both code paths feed the same arguments into CpuFpReferenceSdpa::forward, so
    // results must match to within bit-for-bit tolerance.
    const float tolerance = 0.0f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(deprecatedBundle.oTensor, explicitBundle.oTensor))
        << "Deprecated causal_mask=true should produce identical output to "
           "leftBound=-1, rightBound=0, TOP_LEFT alignment.";
}

TEST(TestSdpaFwdPlanBuilder, DeprecatedCausalMaskBottomRightMatchesExplicitBottomRightBounds)
{
    // The dispatcher maps the deprecated causal_mask_bottom_right=true flag to
    // (leftBound=-1, rightBound=0, BOTTOM_RIGHT). Verify by comparing against an
    // explicit bottom-right window configuration. Use Sq != Skv so that TOP_LEFT
    // and BOTTOM_RIGHT alignments produce different output, catching any bug
    // where the dispatcher forgets to set the alignment to BOTTOM_RIGHT.
    const std::vector<int64_t> qDims = {1, 2, 2, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> explicitBundle(qDims, kDims, vDims, seed);

    // (a) Deprecated causal_mask_bottom_right=true
    auto deprecatedGraphTuple = buildSdpaFwdGraph(deprecatedBundle,
                                                  DataType::FLOAT,
                                                  /*causalMask=*/false,
                                                  /*causalMaskBottomRight=*/true,
                                                  /*leftBound=*/std::nullopt,
                                                  /*rightBound=*/std::nullopt,
                                                  hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& deprecatedGraph = std::get<0>(deprecatedGraphTuple);
    auto [depBin, depErr] = deprecatedGraph->to_binary();
    ASSERT_TRUE(depErr.is_good()) << depErr.get_message();
    const GraphWrapper depWrapper(depBin.data(), depBin.size());

    // (b) Explicit (leftBound=-1, rightBound=0, BOTTOM_RIGHT)
    auto explicitGraphTuple = buildSdpaFwdGraph(explicitBundle,
                                                DataType::FLOAT,
                                                /*causalMask=*/false,
                                                /*causalMaskBottomRight=*/false,
                                                /*leftBound=*/-1,
                                                /*rightBound=*/0,
                                                hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
    auto& explicitGraph = std::get<0>(explicitGraphTuple);
    auto [expBin, expErr] = explicitGraph->to_binary();
    ASSERT_TRUE(expErr.is_good()) << expErr.get_message();
    const GraphWrapper expWrapper(expBin.data(), expBin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    // Execute deprecated-path plan
    {
        auto plan = planBuilder.buildNodePlan(depWrapper, depWrapper.getNode(0));
        const auto* attrs = depWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = deprecatedBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = deprecatedBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = deprecatedBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = deprecatedBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Execute explicit-bounds plan
    {
        auto plan = planBuilder.buildNodePlan(expWrapper, expWrapper.getNode(0));
        const auto* attrs = expWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = explicitBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = explicitBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = explicitBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = explicitBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    const float tolerance = 0.0f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(deprecatedBundle.oTensor, explicitBundle.oTensor))
        << "Deprecated causal_mask_bottom_right=true should produce identical output to "
           "leftBound=-1, rightBound=0, BOTTOM_RIGHT alignment.";
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableFp8RequiresDescale)
{
    // FP8 inputs require q/k/v descales (mirrors AITER's TORCH_CHECK). The dispatcher
    // must reject fp8-without-descale and accept fp8-with-descale.
    const std::vector<int64_t> dims = {1, 2, 4, 8};

    const SdpaFwdPlanBuilder<DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::BFLOAT16>
        fp8Builder;

    {
        auto graph = buildDescaleSdpaFwdGraph(dims,
                                              dims,
                                              dims,
                                              hipdnn_frontend::DataType::FP8_E4M3,
                                              hipdnn_frontend::DataType::BFLOAT16,
                                              /*withDescale=*/false);
        auto [bin, err] = graph->to_binary();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        const GraphWrapper wrapper(bin.data(), bin.size());
        EXPECT_FALSE(fp8Builder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
            << "FP8 inputs without descales must be rejected";
    }

    {
        auto graph = buildDescaleSdpaFwdGraph(dims,
                                              dims,
                                              dims,
                                              hipdnn_frontend::DataType::FP8_E4M3,
                                              hipdnn_frontend::DataType::BFLOAT16,
                                              /*withDescale=*/true);
        auto [bin, err] = graph->to_binary();
        ASSERT_TRUE(err.is_good()) << err.get_message();
        const GraphWrapper wrapper(bin.data(), bin.size());
        EXPECT_TRUE(fp8Builder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
            << "FP8 inputs with q/k/v descales must be accepted";
    }
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsDescaleForNonFp8)
{
    // Non-FP8 inputs must not carry descales; the dispatcher rejects such graphs.
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    auto graph = buildDescaleSdpaFwdGraph(dims,
                                          dims,
                                          dims,
                                          hipdnn_frontend::DataType::FLOAT,
                                          hipdnn_frontend::DataType::FLOAT,
                                          /*withDescale=*/true);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatBuilder;
    EXPECT_FALSE(floatBuilder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
        << "Non-FP8 inputs carrying descales must be rejected";
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsFp8WithSoftmaxDescale)
{
    // The v3 forward kernel supports only q/k/v descales. Softmax/output
    // (de)quantization (descale_s, scale_s/o, amax) is not modeled, so an fp8 graph
    // that additionally carries descale_s must be rejected even though q/k/v descales
    // are present.
    const std::vector<int64_t> dims = {1, 2, 4, 8};
    auto graph = buildDescaleSdpaFwdGraph(dims,
                                          dims,
                                          dims,
                                          hipdnn_frontend::DataType::FP8_E4M3,
                                          hipdnn_frontend::DataType::BFLOAT16,
                                          /*withDescale=*/true,
                                          /*withDescaleS=*/true);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());
    const SdpaFwdPlanBuilder<DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::BFLOAT16>
        fp8Builder;
    EXPECT_FALSE(fp8Builder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()))
        << "fp8 graphs carrying descale_s (softmax quant) must be rejected";
}

TEST(TestSdpaFwdPlan, ExecuteFp8WithDescale)
{
    // Drives the full FP8 path through SdpaFwdPlanBuilder -> SdpaFwdPlan::execute:
    // descale UIDs are read in buildNodePlan and the descale tensors are passed to
    // CpuFpReferenceSdpa. Validates against an independent dequantized-float reference.
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::fp8_e4m3;

    const int64_t batch = 1;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4;
    const int64_t headDim = 8;
    const std::vector<int64_t> dims = {batch, numHeads, seqLen, headDim};

    Tensor<fp8_e4m3> qTensor(dims);
    Tensor<fp8_e4m3> kTensor(dims);
    Tensor<fp8_e4m3> vTensor(dims);
    Tensor<bfloat16> oTensor(dims);

    // Deterministic, exactly-representable fp8 values (small multiples of 1/4).
    auto fillPattern = [&](Tensor<fp8_e4m3>& t, int64_t phase) {
        int64_t flat = 0;
        for(int64_t b = 0; b < batch; ++b)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                for(int64_t s = 0; s < seqLen; ++s)
                {
                    for(int64_t d = 0; d < headDim; ++d)
                    {
                        const float value = static_cast<float>(((flat + phase) % 5) - 2) * 0.25f;
                        t.setHostValue(fp8_e4m3(value), b, h, s, d);
                        ++flat;
                    }
                }
            }
        }
    };
    fillPattern(qTensor, 0);
    fillPattern(kTensor, 1);
    fillPattern(vTensor, 2);

    Tensor<float> descaleQ({1, 1, 1, 1});
    Tensor<float> descaleK({1, 1, 1, 1});
    Tensor<float> descaleV({1, 1, 1, 1});
    descaleQ.fillWithValue(2.0f);
    descaleK.fillWithValue(0.5f);
    descaleV.fillWithValue(1.5f);

    auto graph = buildDescaleSdpaFwdGraph(dims,
                                          dims,
                                          dims,
                                          hipdnn_frontend::DataType::FP8_E4M3,
                                          hipdnn_frontend::DataType::BFLOAT16,
                                          /*withDescale=*/true);
    auto [bin, err] = graph->to_binary();
    ASSERT_TRUE(err.is_good()) << err.get_message();
    const GraphWrapper wrapper(bin.data(), bin.size());

    const SdpaFwdPlanBuilder<DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::FP8_E4M3,
                             DataType::BFLOAT16>
        planBuilder;
    ASSERT_TRUE(planBuilder.isApplicable(wrapper.getNode(0), wrapper.getTensorMap()));
    auto plan = planBuilder.buildNodePlan(wrapper, wrapper.getNode(0));

    const auto* nodeAttributes = wrapper.getNode(0).attributes_as_SdpaAttributes();
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = oTensor.memory().hostData();
    variantPack[nodeAttributes->descale_q_tensor_uid().value()] = descaleQ.memory().hostData();
    variantPack[nodeAttributes->descale_k_tensor_uid().value()] = descaleK.memory().hostData();
    variantPack[nodeAttributes->descale_v_tensor_uid().value()] = descaleV.memory().hostData();
    plan->execute(variantPack);

    // Independent reference: dequantize inputs to float (decode(fp8) * descale) and run
    // the plain (descale-free) reference.
    Tensor<float> qF(dims);
    Tensor<float> kF(dims);
    Tensor<float> vF(dims);
    Tensor<bfloat16> oExpected(dims);
    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            for(int64_t s = 0; s < seqLen; ++s)
            {
                for(int64_t d = 0; d < headDim; ++d)
                {
                    qF.setHostValue(
                        static_cast<float>(qTensor.getHostValue(b, h, s, d)) * 2.0f, b, h, s, d);
                    kF.setHostValue(
                        static_cast<float>(kTensor.getHostValue(b, h, s, d)) * 0.5f, b, h, s, d);
                    vF.setHostValue(
                        static_cast<float>(vTensor.getHostValue(b, h, s, d)) * 1.5f, b, h, s, d);
                }
            }
        }
    }
    CpuFpReferenceSdpa::forward<float, float, float, bfloat16, float>(qF, kF, vF, oExpected);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(oExpected, oTensor))
        << "FP8 plan output does not match the dequantized-input reference";
}
